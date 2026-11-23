# 05 — Agent 3 : Prédiction de Trajectoire sous Blackout Réseau (LSTM / GRU)

## 1. Objectif et Limites de la Baseline

Lors de l'évaluation de la baseline (`docs/03_baseline_evaluation.md`), la gestion des coupures réseau (`is_blackout = 1`, soit 25 épisodes continus de 20 frames / $2{,}0\text{ s}$) reposait sur une **extrapolation linéaire d'ordre 1** à vitesse constante :

$$\hat{\mathbf{P}}_{t+1} = \hat{\mathbf{P}}_t + \mathbf{V}_{t_0} \cdot \Delta t$$

Cette approche s'est révélée insuffisante face aux manœuvres courbes des drones, produisant une erreur **RMSE de `2.8425 m`**, supérieure au seuil critique de collision physique fixé à **`2.0 m`**. En cas de coupure radio, la tour de contrôle perd la localisation fiable du drone muet et ne peut plus garantir la sécurité des appareils voisins.

L'**Agent 3** résout cette vulnérabilité en remplaçant l'extrapolation linéaire par un **réseau de neurones récurrent hybride (LSTM + GRU)** exécuté côté tour de contrôle, capable de modéliser la cinématique non linéaire (courbure, accélération, taux de virage) et de reconstruire la trajectoire en boucle fermée pendant toute la durée du blackout.

---

## 2. Fondements Théoriques et Architecture

### 2.1. Formulation en Déplacement Relatif ($\Delta \mathbf{P}$)

Plutôt que de prédire directement les coordonnées absolues $(tx, ty, tz)$ — qui dépendent de la zone de l'arène ($90\text{ m} \times 90\text{ m} \times 25\text{ m}$) — le modèle est entraîné à prédire le **vecteur déplacement élémentaire** entre deux pas de temps consécutifs ($\Delta t = 0{,}1\text{ s}$) :

$$\Delta \mathbf{P}_{t} = \begin{pmatrix} tx_t - tx_{t-1} \\ ty_t - ty_{t-1} \\ tz_t - tz_{t-1} \end{pmatrix}$$

La position absolue est reconstruite par intégration discrète :

$$\hat{\mathbf{P}}_t = \hat{\mathbf{P}}_{t-1} + \widehat{\Delta \mathbf{P}}_t$$

Cette formulation rend le réseau **invariant par translation** dans l'espace 3D, facilite la convergence et réduit significativement la dérive cumulée.

### 2.2. Représentation Temporelle (Fenêtre Glissante)

À chaque instant $t$, l'entrée du réseau est une séquence temporelle $X_t \in \mathbb{R}^{10 \times 6}$ couvrant la dernière seconde de vol ($W = 10$ frames à $10\text{ Hz}$). Chaque pas de temps contient 6 variables cinématiques normalisées (`StandardScaler`) :

- Les 3 composantes de déplacement local : $(\delta x, \delta y, \delta z)$
- Les 3 composantes du vecteur vitesse : $(v_x, v_y, v_z)$

### 2.3. Architecture Hybride `LSTM + GRU`

Implémentée en **PyTorch** (`scripts/05_agent3_lstm_blackout.py`), l'architecture vise le meilleur compromis entre capacité de mémorisation cinématique et faible latence d'inférence temps réel :

1. **Couche `LSTM` (64 unités)** : Extrait les dépendances cinématiques sur l'ensemble de la fenêtre d'historique ($1{,}0\text{ s}$).
2. **Couche `GRU` (32 unités)** : Compresse la séquence temporelle en un vecteur d'état dynamique avec une empreinte mémoire réduite.
3. **Tête de régression (`Linear(32 → 16)` + `ReLU` + `Linear(16 → 3)`)** : Projette l'état latent vers le vecteur déplacement prédit $(\widehat{\Delta x}, \widehat{\Delta y}, \widehat{\Delta z})$.

### 2.4. Inférence Autorégressive en Boucle Fermée

Durant un blackout (`is_blackout = 1`), le modèle n'accède **jamais** aux coordonnées réelles du dataset :

- À la première frame de coupure $t_0$, le modèle lit les 10 dernières frames réelles reçues avant la perte de signal et prédit $\widehat{\Delta \mathbf{P}}_{t_0}$.
- Pour les 19 frames suivantes ($t_0+1 \dots t_0+19$), chaque prédiction $(\widehat{\Delta \mathbf{P}}_k, \widehat{\mathbf{V}}_k = \widehat{\Delta \mathbf{P}}_k / \Delta t)$ est réinjectée en fin de fenêtre glissante.
- Dès le rétablissement du lien (`is_blackout = 0`), la télémétrie réelle reprend immédiatement la main.

---

## 3. Protocole Expérimental (Zero Data Leakage)

Afin de garantir une évaluation strictement réaliste et sans fuite de données entre l'entraînement et le test, le dataset (`43 573` enregistrements, `250` drones) a été partitionné **par identifiant de drone (`drone_id`)** :

| Ensemble                   | Nombre de Drones |           Séquences / Frames           | Rôle                                                                                     |
| :------------------------- | :--------------: | :------------------------------------: | :--------------------------------------------------------------------------------------- |
| **Entraînement (`Train`)** | 200 drones sains |     `32 752` séquences `(10 × 6)`      | Apprentissage de la cinématique de vol (`MSELoss`, `Adam`, `lr=1e-3`)                    |
| **Validation (`Val`)**     | 25 drones sains  |      `4 048` séquences `(10 × 6)`      | Sélection du meilleur modèle et régulation du taux d'apprentissage (`ReduceLROnPlateau`) |
| **Test Blackout (`Test`)** | 25 drones coupés | `500` frames (25 épisodes × 20 frames) | Évaluation autorégressive stricte sur des drones **jamais vus** à l'entraînement         |

- **Normalisation** : Les objets `StandardScaler` (`agent3_scaler_x.pkl`, `agent3_scaler_y.pkl`) sont ajustés **exclusivement sur le jeu d'entraînement**.
- **Matériel** : Entraînement et inférence exécutés sur accélérateur Apple Silicon (**MPS**) en **`64,95 s`** sur 40 époques (`Train MSE` finale : `0.043007`, `Val MSE` finale : `0.044241`, sans surapprentissage).

---

## 4. Résultats Quantitatifs : Baseline vs Agent 3

Les métriques enregistrées dans `results/agent3_blackout_metrics.json` démontrent un gain majeur par rapport à la baseline linéaire :

| Métrique                                  | Baseline (Linéaire) | Agent 3 (LSTM/GRU)  |        Évolution / Gain         |
| :---------------------------------------- | :-----------------: | :-----------------: | :-----------------------------: |
| **Épisodes de Blackout évalués**          |   25 (500 frames)   |   25 (500 frames)   |            Identique            |
| **RMSE Localisation sous Blackout**       |     `2.8425 m`      |   **`1.1068 m`**    |  **`-61,06 %` (`-1.7357 m`)**   |
| **Respect du seuil critique (`< 2.0 m`)** | Dépassé (`2.84 m`)  | Respecté (`1.11 m`) |          **Sécurisé**           |
| **Latence moyenne d'inférence**           |          —          |   **`1.9964 ms`**   | Compatible temps réel (`10 Hz`) |
| **Temps d'entraînement total**            |          —          |  `64.95 s` (`mps`)  |     Rapide et reproductible     |

---

## 5. Analyse et Intégration dans l'Architecture Multi-Agents

1. **Passage sous le seuil de collision critique** : La réduction de l'erreur de `2.8425 m` à `1.1068 m` permet à la tour de contrôle de maintenir une estimation fiable de la position du drone aveugle pendant toute la durée de la coupure ($2{,}0\text{ s}$).
2. **Contrainte temps réel respectée** : Avec une latence moyenne de **`1.9964 ms`** par prédiction, l'Agent 3 consomme moins de **2 %** du budget temporel d'une frame à $10\text{ Hz}$ ($100\text{ ms}$), laissant une large marge pour l'Agent 2 (Cybersécurité) et l'Agent 1 (Priorisation / Grille 3D).
3. **Rôle dans la prévention active des collisions (Gazebo)** : En simulation temps réel, le drone en blackout ne pouvant ni émettre ni recevoir d'ordres, la tour de contrôle utilise la trajectoire prédite par l'Agent 3 pour créer une bulle de sécurité prioritaire (`P0 CRITIQUE`) et dévier dynamiquement les drones voisins connectés autour de cette zone.

---

## 6. Artefacts Générés

- **Script d'entraînement et d'évaluation** : `scripts/05_agent3_lstm_blackout.py`
- **Poids du modèle entraîné** : `models/agent3_lstm_blackout.pt`
- **Normaliseurs sauvegardés** : `models/agent3_scaler_x.pkl`, `models/agent3_scaler_y.pkl`
- **Métriques JSON** : `results/agent3_blackout_metrics.json`
